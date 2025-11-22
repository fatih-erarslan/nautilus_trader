use axum::{
    extract::{Query, State}, 
    routing::get, 
    Router, 
    Json,
    http::StatusCode
};
use chrono::{DateTime, Utc, Duration};
use dashmap::DashMap;
use governor::{Quota, RateLimiter};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration as StdDuration};
use tokio::time::sleep;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer, compression::CompressionLayer};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketData {
    symbol: String,
    price: f64,
    volume_24h: f64,
    change_24h: f64,
    high_24h: f64,
    low_24h: f64,
    quote_volume: f64,
    timestamp: DateTime<Utc>,
    source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TechnicalIndicators {
    atr: f64,
    rsi: f64,
    volatility_score: f64,
    volume_profile: f64,
    fibonacci_level: Option<f64>,
    support_resistance: (f64, f64), // (support, resistance)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WhaleActivity {
    large_orders_count: u32,
    whale_volume_ratio: f64,
    order_book_depth: f64,
    unusual_activity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PairIntelligence {
    symbol: String,
    market_data: MarketData,
    technical_indicators: TechnicalIndicators,
    whale_activity: WhaleActivity,
    ai_confidence: f64,
    profitability_score: f64,
    recommendation_score: f64,
    risk_level: String, // "LOW", "MEDIUM", "HIGH"
    last_updated: DateTime<Utc>,
    available_exchanges: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinanceTicker {
    symbol: String,
    #[serde(rename = "lastPrice")]
    price: String,
    #[serde(rename = "priceChangePercent")]
    price_change_percent: String,
    #[serde(rename = "highPrice")]
    high_price: String,
    #[serde(rename = "lowPrice")]
    low_price: String,
    volume: String,
    #[serde(rename = "quoteVolume")]
    quote_volume: String,
    count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OkxTicker {
    #[serde(rename = "instId")]
    symbol: String,
    last: String,
    #[serde(rename = "lastSz")]
    last_size: String,
    #[serde(rename = "askPx")]
    ask_price: String,
    #[serde(rename = "bidPx")]
    bid_price: String,
    #[serde(rename = "open24h")]
    open_24h: String,
    #[serde(rename = "high24h")]
    high_24h: String,
    #[serde(rename = "low24h")]
    low_24h: String,
    #[serde(rename = "volCcy24h")]
    quote_volume_24h: String,
    #[serde(rename = "vol24h")]
    base_volume_24h: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OkxResponse {
    code: String,
    msg: String,
    data: Vec<OkxTicker>,
}

type IntelligenceStore = Arc<DashMap<String, PairIntelligence>>;
type RateLimiterType = Arc<RateLimiter<governor::state::direct::NotKeyed, governor::state::InMemoryState, governor::clock::DefaultClock>>;

#[derive(Clone)]
struct AppState {
    intelligence_store: IntelligenceStore,
    client: Client,
    binance_limiter: RateLimiterType,
    okx_limiter: RateLimiterType,
}

#[derive(Deserialize)]
struct IntelligenceQuery {
    limit: Option<u32>,
    min_volume: Option<f64>,
    exchange: Option<String>,
    sort_by: Option<String>, // "profitability", "volatility", "ai_confidence", "volume"
}

#[derive(Deserialize)]
struct PairQuery {
    symbol: String,
    timeframe: Option<String>,
}

async fn fetch_binance_data(client: &Client, limiter: &RateLimiterType) -> Result<Vec<MarketData>, Box<dyn std::error::Error + Send + Sync>> {
    limiter.until_ready().await;
    
    debug!("Fetching ALL Binance market data...");
    let url = "https://api.binance.com/api/v3/ticker/24hr";
    
    let response = client.get(url)
        .timeout(StdDuration::from_secs(30))
        .send()
        .await?;
    
    if !response.status().is_success() {
        return Err(format!("Binance API error: {}", response.status()).into());
    }
    
    let tickers: Vec<BinanceTicker> = response.json().await?;
    debug!("Received {} tickers from Binance", tickers.len());
    
    // Filter for liquid USDT pairs with significant volume
    let market_data = tickers.into_iter()
        .filter(|ticker| {
            let volume: f64 = ticker.quote_volume.parse().unwrap_or(0.0);
            let price: f64 = ticker.price.parse().unwrap_or(0.0);
            ticker.symbol.ends_with("USDT") && volume > 100000.0 && price > 0.0001
        })
        .map(|ticker| MarketData {
            symbol: ticker.symbol.clone(),
            price: ticker.price.parse().unwrap_or(0.0),
            volume_24h: ticker.volume.parse().unwrap_or(0.0),
            change_24h: ticker.price_change_percent.parse().unwrap_or(0.0),
            high_24h: ticker.high_price.parse().unwrap_or(0.0),
            low_24h: ticker.low_price.parse().unwrap_or(0.0),
            quote_volume: ticker.quote_volume.parse().unwrap_or(0.0),
            timestamp: Utc::now(),
            source: "binance".to_string(),
        })
        .collect();
    
    Ok(market_data)
}

async fn fetch_okx_data(client: &Client, limiter: &RateLimiterType) -> Result<Vec<MarketData>, Box<dyn std::error::Error + Send + Sync>> {
    limiter.until_ready().await;
    
    debug!("Fetching ALL OKX market data...");
    let url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT";
    
    let response = client.get(url)
        .timeout(StdDuration::from_secs(30))
        .send()
        .await?;
    
    if !response.status().is_success() {
        return Err(format!("OKX API error: {}", response.status()).into());
    }
    
    let okx_response: OkxResponse = response.json().await?;
    debug!("Received {} tickers from OKX", okx_response.data.len());
    
    if okx_response.code != "0" {
        return Err(format!("OKX API error: {}", okx_response.msg).into());
    }
    
    // Filter for liquid USDT pairs
    let market_data = okx_response.data.into_iter()
        .filter(|ticker| {
            let volume: f64 = ticker.quote_volume_24h.parse().unwrap_or(0.0);
            let price: f64 = ticker.last.parse().unwrap_or(0.0);
            ticker.symbol.ends_with("-USDT") && volume > 50000.0 && price > 0.0001
        })
        .map(|ticker| {
            let open: f64 = ticker.open_24h.parse().unwrap_or(0.0);
            let last: f64 = ticker.last.parse().unwrap_or(0.0);
            let change_24h = if open > 0.0 {
                ((last - open) / open) * 100.0
            } else { 0.0 };
            
            MarketData {
                symbol: ticker.symbol.replace("-", ""), // Convert OKX format to Binance format
                price: last,
                volume_24h: ticker.base_volume_24h.parse().unwrap_or(0.0),
                change_24h,
                high_24h: ticker.high_24h.parse().unwrap_or(0.0),
                low_24h: ticker.low_24h.parse().unwrap_or(0.0),
                quote_volume: ticker.quote_volume_24h.parse().unwrap_or(0.0),
                timestamp: Utc::now(),
                source: "okx".to_string(),
            }
        })
        .collect();
    
    Ok(market_data)
}

fn calculate_technical_indicators(market_data: &MarketData) -> TechnicalIndicators {
    // Simplified calculations - in production, you'd use historical data
    let price_range = market_data.high_24h - market_data.low_24h;
    let atr = price_range / market_data.price; // Simplified ATR
    
    // Volatility based on 24h price movement
    let volatility_score = (market_data.change_24h.abs() / 10.0).min(10.0);
    
    // Volume profile based on quote volume
    let volume_profile = if market_data.quote_volume > 1_000_000.0 {
        10.0
    } else if market_data.quote_volume > 100_000.0 {
        5.0 + (market_data.quote_volume / 200_000.0)
    } else {
        market_data.quote_volume / 50_000.0
    };
    
    // Simplified RSI based on price change
    let rsi = 50.0 + (market_data.change_24h / 2.0).min(40.0).max(-40.0);
    
    TechnicalIndicators {
        atr,
        rsi,
        volatility_score,
        volume_profile,
        fibonacci_level: None, // Would need historical data
        support_resistance: (market_data.low_24h, market_data.high_24h),
    }
}

fn detect_whale_activity(market_data: &MarketData) -> WhaleActivity {
    // Simplified whale detection based on volume patterns
    let volume_ratio = market_data.quote_volume / 1_000_000.0;
    let price_impact = market_data.change_24h.abs();
    
    let whale_volume_ratio = if volume_ratio > 10.0 && price_impact > 5.0 {
        0.8
    } else if volume_ratio > 5.0 {
        0.5
    } else {
        0.2
    };
    
    let unusual_activity_score = (volume_ratio * price_impact / 10.0).min(10.0);
    
    WhaleActivity {
        large_orders_count: (volume_ratio as u32).min(100),
        whale_volume_ratio,
        order_book_depth: volume_ratio,
        unusual_activity_score,
    }
}

fn calculate_ai_confidence(technical: &TechnicalIndicators, whale: &WhaleActivity) -> f64 {
    // AI confidence based on multiple factors
    let volatility_confidence = if technical.volatility_score > 3.0 && technical.volatility_score < 8.0 {
        0.8 // Good volatility for trading
    } else {
        0.4
    };
    
    let volume_confidence = (technical.volume_profile / 10.0).min(1.0);
    let whale_confidence = whale.whale_volume_ratio;
    
    (volatility_confidence + volume_confidence + whale_confidence) / 3.0
}

fn calculate_profitability_score(market_data: &MarketData, technical: &TechnicalIndicators) -> f64 {
    let volume_factor = (market_data.quote_volume / 1_000_000.0).min(5.0);
    let volatility_factor = technical.volatility_score / 2.0;
    let change_factor = market_data.change_24h.abs() / 5.0;
    
    (volume_factor + volatility_factor + change_factor) / 3.0 * 10.0
}

fn generate_pair_intelligence(market_data: MarketData, available_exchanges: Vec<String>) -> PairIntelligence {
    let technical_indicators = calculate_technical_indicators(&market_data);
    let whale_activity = detect_whale_activity(&market_data);
    let ai_confidence = calculate_ai_confidence(&technical_indicators, &whale_activity);
    let profitability_score = calculate_profitability_score(&market_data, &technical_indicators);
    
    // Overall recommendation score (0-10)
    let recommendation_score = (ai_confidence * 4.0) + 
                              (profitability_score / 2.0) + 
                              (technical_indicators.volume_profile / 5.0);
    
    let risk_level = if recommendation_score > 7.0 {
        "LOW".to_string()
    } else if recommendation_score > 4.0 {
        "MEDIUM".to_string()
    } else {
        "HIGH".to_string()
    };
    
    PairIntelligence {
        symbol: market_data.symbol.clone(),
        market_data,
        technical_indicators,
        whale_activity,
        ai_confidence,
        profitability_score,
        recommendation_score,
        risk_level,
        last_updated: Utc::now(),
        available_exchanges,
    }
}

async fn update_intelligence_data(state: AppState) {
    let mut interval = tokio::time::interval(StdDuration::from_secs(30));
    
    loop {
        interval.tick().await;
        
        info!("🧠 Updating CWTS Intelligence data from all exchanges...");
        
        let mut all_pairs: HashMap<String, Vec<String>> = HashMap::new();
        
        // Fetch from Binance
        match fetch_binance_data(&state.client, &state.binance_limiter).await {
            Ok(data) => {
                info!("📊 Successfully fetched {} liquid pairs from Binance", data.len());
                for item in data {
                    let intelligence = generate_pair_intelligence(item.clone(), vec!["binance".to_string()]);
                    all_pairs.entry(item.symbol.clone())
                        .or_insert_with(Vec::new)
                        .push("binance".to_string());
                    state.intelligence_store.insert(format!("binance_{}", item.symbol), intelligence);
                }
            },
            Err(e) => error!("❌ Failed to fetch Binance data: {}", e),
        }
        
        // Fetch from OKX
        match fetch_okx_data(&state.client, &state.okx_limiter).await {
            Ok(data) => {
                info!("📊 Successfully fetched {} liquid pairs from OKX", data.len());
                for item in data {
                    let intelligence = generate_pair_intelligence(item.clone(), vec!["okx".to_string()]);
                    all_pairs.entry(item.symbol.clone())
                        .or_insert_with(Vec::new)
                        .push("okx".to_string());
                    state.intelligence_store.insert(format!("okx_{}", item.symbol), intelligence);
                }
            },
            Err(e) => error!("❌ Failed to fetch OKX data: {}", e),
        }
        
        // Update cross-exchange availability
        for (symbol, exchanges) in all_pairs {
            if exchanges.len() > 1 {
                debug!("🔗 {} available on: {:?}", symbol, exchanges);
                
                // Update both entries with cross-exchange availability
                for exchange in &exchanges {
                    let key = format!("{}_{}", exchange, symbol);
                    if let Some(mut intelligence) = state.intelligence_store.get_mut(&key) {
                        intelligence.available_exchanges = exchanges.clone();
                    }
                }
            }
        }
        
        info!("🎯 Intelligence update completed. Total analyzed pairs: {}", state.intelligence_store.len());
    }
}

async fn get_intelligence_recommendations(
    State(state): State<AppState>,
    Query(params): Query<IntelligenceQuery>
) -> Result<Json<Vec<PairIntelligence>>, StatusCode> {
    let limit = params.limit.unwrap_or(20).min(100) as usize;
    let min_volume = params.min_volume.unwrap_or(100_000.0);
    let sort_by = params.sort_by.unwrap_or_else(|| "recommendation_score".to_string());
    
    let mut recommendations: Vec<PairIntelligence> = state.intelligence_store
        .iter()
        .filter(|entry| {
            let intelligence = entry.value();
            intelligence.market_data.quote_volume >= min_volume &&
            intelligence.recommendation_score > 2.0
        })
        .map(|entry| entry.value().clone())
        .collect();
    
    // Sort by specified criteria
    match sort_by.as_str() {
        "profitability" => recommendations.sort_by(|a, b| b.profitability_score.partial_cmp(&a.profitability_score).unwrap()),
        "volatility" => recommendations.sort_by(|a, b| b.technical_indicators.volatility_score.partial_cmp(&a.technical_indicators.volatility_score).unwrap()),
        "ai_confidence" => recommendations.sort_by(|a, b| b.ai_confidence.partial_cmp(&a.ai_confidence).unwrap()),
        "volume" => recommendations.sort_by(|a, b| b.market_data.quote_volume.partial_cmp(&a.market_data.quote_volume).unwrap()),
        _ => recommendations.sort_by(|a, b| b.recommendation_score.partial_cmp(&a.recommendation_score).unwrap()),
    }
    
    recommendations.truncate(limit);
    
    if recommendations.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }
    
    Ok(Json(recommendations))
}

async fn get_pair_analysis(
    State(state): State<AppState>,
    Query(params): Query<PairQuery>
) -> Result<Json<PairIntelligence>, StatusCode> {
    // Look for the pair in both exchanges
    if let Some(binance_data) = state.intelligence_store.get(&format!("binance_{}", params.symbol)) {
        return Ok(Json(binance_data.clone()));
    }
    
    if let Some(okx_data) = state.intelligence_store.get(&format!("okx_{}", params.symbol)) {
        return Ok(Json(okx_data.clone()));
    }
    
    Err(StatusCode::NOT_FOUND)
}

async fn health_check() -> &'static str {
    "🧠 CWTS Intelligence Server - Online"
}

async fn get_system_stats(
    State(state): State<AppState>
) -> Json<HashMap<String, serde_json::Value>> {
    let total_pairs = state.intelligence_store.len();
    let binance_pairs = state.intelligence_store.iter().filter(|entry| entry.key().starts_with("binance_")).count();
    let okx_pairs = state.intelligence_store.iter().filter(|entry| entry.key().starts_with("okx_")).count();
    
    let high_confidence_pairs = state.intelligence_store.iter()
        .filter(|entry| entry.value().ai_confidence > 0.7)
        .count();
    
    let mut stats = HashMap::new();
    stats.insert("total_pairs".to_string(), serde_json::Value::Number(total_pairs.into()));
    stats.insert("binance_pairs".to_string(), serde_json::Value::Number(binance_pairs.into()));
    stats.insert("okx_pairs".to_string(), serde_json::Value::Number(okx_pairs.into()));
    stats.insert("high_confidence_pairs".to_string(), serde_json::Value::Number(high_confidence_pairs.into()));
    stats.insert("last_updated".to_string(), serde_json::Value::String(Utc::now().to_rfc3339()));
    stats.insert("status".to_string(), serde_json::Value::String("analyzing".to_string()));
    
    Json(stats)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("cwts_intelligence_server=info".parse()?))
        .init();
    
    info!("🚀 Starting CWTS Intelligence Server...");
    
    // Initialize HTTP client
    let client = Client::builder()
        .timeout(StdDuration::from_secs(60))
        .user_agent("CWTS-Intelligence/0.1.0")
        .build()?;
    
    // Create rate limiters (Binance: 1200/min, OKX: 20/sec)
    let binance_limiter = Arc::new(RateLimiter::direct(
        Quota::per_minute(std::num::NonZeroU32::new(1200).unwrap())
    ));
    let okx_limiter = Arc::new(RateLimiter::direct(
        Quota::per_second(std::num::NonZeroU32::new(20).unwrap())
    ));
    
    // Initialize intelligence store
    let intelligence_store = Arc::new(DashMap::new());
    
    let state = AppState {
        intelligence_store: intelligence_store.clone(),
        client,
        binance_limiter,
        okx_limiter,
    };
    
    // Start background intelligence analysis
    let analysis_state = state.clone();
    tokio::spawn(update_intelligence_data(analysis_state));
    
    // Give it a moment to fetch initial data
    info!("⏳ Loading initial intelligence data...");
    sleep(StdDuration::from_secs(8)).await;
    
    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/intelligence", get(get_intelligence_recommendations))
        .route("/api/pair-analysis", get(get_pair_analysis))
        .route("/api/stats", get(get_system_stats))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(CorsLayer::permissive())
        )
        .with_state(state);
    
    // Start server
    let port = std::env::var("PORT").unwrap_or_else(|_| "8011".to_string());
    let addr = format!("0.0.0.0:{}", port);
    
    info!("🧠 CWTS Intelligence Server starting on {}", addr);
    info!("📊 Endpoints:");
    info!("  • Health: http://{}/health", addr);
    info!("  • Intelligence: http://{}/api/intelligence", addr);
    info!("  • Pair Analysis: http://{}/api/pair-analysis?symbol=BTCUSDT", addr);
    info!("  • Stats: http://{}/api/stats", addr);
    info!("🔗 Exchanges: Binance (ALL liquid pairs), OKX (ALL liquid pairs)");
    info!("📈 Features: ATR, RSI, Volatility, Volume, Whale Detection, AI Confidence");
    
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}