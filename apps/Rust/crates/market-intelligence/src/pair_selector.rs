use crate::*;
use std::collections::{HashMap, HashSet};

pub struct PairSelector {
    min_volume_24h: f64,
    min_market_cap: f64,
    max_spread: f64,
    supported_exchanges: HashSet<String>,
    excluded_pairs: HashSet<String>,
    preferred_base_currencies: HashSet<String>,
}

impl PairSelector {
    pub fn new() -> Self {
        let mut supported_exchanges = HashSet::new();
        supported_exchanges.insert("binance".to_string());
        supported_exchanges.insert("coinbase".to_string());
        supported_exchanges.insert("kraken".to_string());
        supported_exchanges.insert("kucoin".to_string());
        
        let mut preferred_bases = HashSet::new();
        preferred_bases.insert("USDT".to_string());
        preferred_bases.insert("USDC".to_string());
        preferred_bases.insert("BTC".to_string());
        preferred_bases.insert("ETH".to_string());
        
        let mut excluded = HashSet::new();
        excluded.insert("LUNA/USDT".to_string()); // Historical issues
        excluded.insert("UST/USDT".to_string());
        
        Self {
            min_volume_24h: 1_000_000.0, // $1M minimum volume
            min_market_cap: 10_000_000.0, // $10M minimum market cap
            max_spread: 0.002, // 0.2% max spread
            supported_exchanges,
            excluded_pairs,
            preferred_base_currencies: preferred_bases,
        }
    }
    
    pub async fn get_candidate_pairs(&self) -> Result<Vec<CandidatePair>, IntelligenceError> {
        // Fetch market data from multiple sources
        let binance_pairs = self.fetch_binance_pairs().await?;
        let coinbase_pairs = self.fetch_coinbase_pairs().await?;
        
        // Combine and deduplicate
        let mut all_pairs = HashMap::new();
        
        for pair in binance_pairs {
            all_pairs.insert(pair.symbol.clone(), pair);
        }
        
        for pair in coinbase_pairs {
            // Merge data if pair already exists
            if let Some(existing) = all_pairs.get_mut(&pair.symbol) {
                existing.merge_data(&pair);
            } else {
                all_pairs.insert(pair.symbol.clone(), pair);
            }
        }
        
        // Filter pairs
        let filtered_pairs: Vec<_> = all_pairs.into_values()
            .filter(|pair| self.meets_criteria(pair))
            .collect();
        
        // Sort by attractiveness score
        let mut scored_pairs = filtered_pairs;
        scored_pairs.sort_by(|a, b| b.attractiveness_score.partial_cmp(&a.attractiveness_score).unwrap());
        
        Ok(scored_pairs)
    }
    
    pub async fn select_best_pairs(&self, count: usize) -> Result<Vec<String>, IntelligenceError> {
        let candidates = self.get_candidate_pairs().await?;
        
        // Apply additional selection logic
        let selected = candidates.into_iter()
            .take(count)
            .map(|pair| pair.symbol)
            .collect();
        
        Ok(selected)
    }
    
    async fn fetch_binance_pairs(&self) -> Result<Vec<CandidatePair>, IntelligenceError> {
        let client = reqwest::Client::new();
        
        // Fetch 24hr ticker data
        let url = "https://api.binance.com/api/v3/ticker/24hr";
        let response: Vec<BinanceTicker> = client
            .get(url)
            .send()
            .await?
            .json()
            .await?;
        
        let mut pairs = vec![];
        
        for ticker in response {
            if self.is_preferred_base(&ticker.symbol) {
                let pair = CandidatePair {
                    symbol: ticker.symbol.clone(),
                    exchange: "binance".to_string(),
                    volume_24h: ticker.volume.parse().unwrap_or(0.0),
                    price_change_24h: ticker.price_change_percent.parse().unwrap_or(0.0),
                    last_price: ticker.last_price.parse().unwrap_or(0.0),
                    bid_price: ticker.bid_price.parse().unwrap_or(0.0),
                    ask_price: ticker.ask_price.parse().unwrap_or(0.0),
                    market_cap: 0.0, // Would need separate API call
                    attractiveness_score: 0.0,
                };
                pairs.push(pair);
            }
        }
        
        // Calculate attractiveness scores
        for pair in &mut pairs {
            pair.attractiveness_score = self.calculate_attractiveness_score(pair);
        }
        
        Ok(pairs)
    }
    
    async fn fetch_coinbase_pairs(&self) -> Result<Vec<CandidatePair>, IntelligenceError> {
        let client = reqwest::Client::new();
        
        // Fetch products from Coinbase Pro
        let url = "https://api.exchange.coinbase.com/products";
        let response: Vec<CoinbaseProduct> = client
            .get(url)
            .send()
            .await?
            .json()
            .await?;
        
        let mut pairs = vec![];
        
        for product in response {
            if product.status == "online" && self.is_preferred_base(&product.id) {
                // Get 24hr stats for each product
                if let Ok(stats) = self.fetch_coinbase_stats(&product.id).await {
                    let pair = CandidatePair {
                        symbol: product.id.clone(),
                        exchange: "coinbase".to_string(),
                        volume_24h: stats.volume.parse().unwrap_or(0.0),
                        price_change_24h: ((stats.last.parse::<f64>().unwrap_or(0.0) - 
                                           stats.open.parse::<f64>().unwrap_or(0.0)) / 
                                           stats.open.parse::<f64>().unwrap_or(1.0)) * 100.0,
                        last_price: stats.last.parse().unwrap_or(0.0),
                        bid_price: 0.0, // Would need order book data
                        ask_price: 0.0,
                        market_cap: 0.0,
                        attractiveness_score: 0.0,
                    };
                    pairs.push(pair);
                }
            }
        }
        
        // Calculate attractiveness scores
        for pair in &mut pairs {
            pair.attractiveness_score = self.calculate_attractiveness_score(pair);
        }
        
        Ok(pairs)
    }
    
    async fn fetch_coinbase_stats(&self, product_id: &str) -> Result<CoinbaseStats, IntelligenceError> {
        let client = reqwest::Client::new();
        let url = format!("https://api.exchange.coinbase.com/products/{}/stats", product_id);
        
        let stats = client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;
        
        Ok(stats)
    }
    
    fn meets_criteria(&self, pair: &CandidatePair) -> bool {
        // Volume check
        if pair.volume_24h < self.min_volume_24h {
            return false;
        }
        
        // Spread check
        if pair.bid_price > 0.0 && pair.ask_price > 0.0 {
            let spread = (pair.ask_price - pair.bid_price) / pair.bid_price;
            if spread > self.max_spread {
                return false;
            }
        }
        
        // Excluded pairs check
        if self.excluded_pairs.contains(&pair.symbol) {
            return false;
        }
        
        // Supported exchange check
        if !self.supported_exchanges.contains(&pair.exchange) {
            return false;
        }
        
        true
    }
    
    fn is_preferred_base(&self, symbol: &str) -> bool {
        for base in &self.preferred_base_currencies {
            if symbol.ends_with(base) {
                return true;
            }
        }
        false
    }
    
    fn calculate_attractiveness_score(&self, pair: &CandidatePair) -> f64 {
        let mut score = 0.0;
        
        // Volume factor (higher volume = more attractive)
        let volume_score = (pair.volume_24h / 10_000_000.0).min(1.0); // Normalize to 10M volume
        score += volume_score * 0.3;
        
        // Volatility factor (moderate volatility preferred)
        let volatility = pair.price_change_24h.abs() / 100.0;
        let volatility_score = if volatility > 0.02 && volatility < 0.15 {
            1.0 - (volatility - 0.08).abs() / 0.07 // Peak at 8% volatility
        } else {
            0.0
        };
        score += volatility_score * 0.3;
        
        // Price level factor (prefer mid-range prices for better precision)
        let price_score = if pair.last_price > 0.01 && pair.last_price < 1000.0 {
            1.0
        } else if pair.last_price >= 1000.0 {
            0.7
        } else {
            0.5
        };
        score += price_score * 0.2;
        
        // Exchange preference
        let exchange_score = match pair.exchange.as_str() {
            "binance" => 1.0,
            "coinbase" => 0.9,
            "kraken" => 0.8,
            _ => 0.5,
        };
        score += exchange_score * 0.2;
        
        score.min(1.0)
    }
}

#[derive(Debug, Clone)]
pub struct CandidatePair {
    pub symbol: String,
    pub exchange: String,
    pub volume_24h: f64,
    pub price_change_24h: f64,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub market_cap: f64,
    pub attractiveness_score: f64,
}

impl CandidatePair {
    pub fn merge_data(&mut self, other: &CandidatePair) {
        // Merge data from multiple exchanges
        self.volume_24h += other.volume_24h;
        
        // Use weighted average for prices
        let total_vol = self.volume_24h + other.volume_24h;
        if total_vol > 0.0 {
            self.last_price = (self.last_price * self.volume_24h + 
                              other.last_price * other.volume_24h) / total_vol;
        }
        
        // Take the better bid/ask if available
        if other.bid_price > self.bid_price {
            self.bid_price = other.bid_price;
        }
        if other.ask_price < self.ask_price && other.ask_price > 0.0 {
            self.ask_price = other.ask_price;
        }
    }
}

// API response structures
#[derive(Debug, serde::Deserialize)]
struct BinanceTicker {
    symbol: String,
    #[serde(rename = "priceChangePercent")]
    price_change_percent: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "bidPrice")]
    bid_price: String,
    #[serde(rename = "askPrice")]
    ask_price: String,
    volume: String,
}

#[derive(Debug, serde::Deserialize)]
struct CoinbaseProduct {
    id: String,
    status: String,
    #[serde(rename = "base_currency")]
    base_currency: String,
    #[serde(rename = "quote_currency")]
    quote_currency: String,
}

#[derive(Debug, serde::Deserialize)]
struct CoinbaseStats {
    open: String,
    high: String,
    low: String,
    volume: String,
    last: String,
}

// Advanced pair selection strategies
pub struct AdvancedPairSelector {
    base_selector: PairSelector,
    momentum_weight: f64,
    mean_reversion_weight: f64,
    correlation_threshold: f64,
}

impl AdvancedPairSelector {
    pub fn new() -> Self {
        Self {
            base_selector: PairSelector::new(),
            momentum_weight: 0.6,
            mean_reversion_weight: 0.4,
            correlation_threshold: 0.7,
        }
    }
    
    pub async fn select_pairs_by_strategy(&self, strategy: SelectionStrategy, count: usize) -> Result<Vec<String>, IntelligenceError> {
        let candidates = self.base_selector.get_candidate_pairs().await?;
        
        match strategy {
            SelectionStrategy::Momentum => self.select_momentum_pairs(candidates, count),
            SelectionStrategy::MeanReversion => self.select_mean_reversion_pairs(candidates, count),
            SelectionStrategy::LowCorrelation => self.select_low_correlation_pairs(candidates, count).await,
            SelectionStrategy::Balanced => self.select_balanced_portfolio(candidates, count).await,
        }
    }
    
    fn select_momentum_pairs(&self, mut candidates: Vec<CandidatePair>, count: usize) -> Result<Vec<String>, IntelligenceError> {
        // Sort by price momentum
        candidates.sort_by(|a, b| b.price_change_24h.partial_cmp(&a.price_change_24h).unwrap());
        
        // Select top momentum pairs
        Ok(candidates.into_iter()
            .take(count)
            .map(|p| p.symbol)
            .collect())
    }
    
    fn select_mean_reversion_pairs(&self, mut candidates: Vec<CandidatePair>, count: usize) -> Result<Vec<String>, IntelligenceError> {
        // Sort by negative momentum (most oversold)
        candidates.sort_by(|a, b| a.price_change_24h.partial_cmp(&b.price_change_24h).unwrap());
        
        // Select most oversold pairs
        Ok(candidates.into_iter()
            .take(count)
            .map(|p| p.symbol)
            .collect())
    }
    
    async fn select_low_correlation_pairs(&self, candidates: Vec<CandidatePair>, count: usize) -> Result<Vec<String>, IntelligenceError> {
        // This would implement correlation analysis
        // For now, return diverse selection by market cap tiers
        let mut selected = vec![];
        let mut large_cap = 0;
        let mut mid_cap = 0;
        let mut small_cap = 0;
        
        for candidate in candidates {
            if selected.len() >= count {
                break;
            }
            
            // Simple diversification by assumed market cap
            let is_major = candidate.symbol.contains("BTC") || 
                          candidate.symbol.contains("ETH") ||
                          candidate.symbol.contains("BNB");
            
            if is_major && large_cap < count / 3 {
                selected.push(candidate.symbol);
                large_cap += 1;
            } else if !is_major && candidate.volume_24h > 5_000_000.0 && mid_cap < count / 3 {
                selected.push(candidate.symbol);
                mid_cap += 1;
            } else if !is_major && small_cap < count / 3 {
                selected.push(candidate.symbol);
                small_cap += 1;
            }
        }
        
        Ok(selected)
    }
    
    async fn select_balanced_portfolio(&self, candidates: Vec<CandidatePair>, count: usize) -> Result<Vec<String>, IntelligenceError> {
        // Implement modern portfolio theory concepts
        // For now, balance between momentum and mean reversion
        
        let momentum_count = (count as f64 * self.momentum_weight) as usize;
        let mean_reversion_count = count - momentum_count;
        
        let momentum_pairs = self.select_momentum_pairs(candidates.clone(), momentum_count)?;
        let mean_reversion_pairs = self.select_mean_reversion_pairs(candidates, mean_reversion_count)?;
        
        let mut balanced = momentum_pairs;
        balanced.extend(mean_reversion_pairs);
        
        Ok(balanced)
    }
}

#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    Momentum,
    MeanReversion,
    LowCorrelation,
    Balanced,
}