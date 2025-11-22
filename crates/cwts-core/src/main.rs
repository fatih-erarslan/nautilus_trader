use axum::{
    extract::{Query, State}, 
    routing::get, 
    Router, 
    Json,
    http::StatusCode
};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::time::sleep;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer, compression::CompressionLayer};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

// ATS-CP Structures
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ATSCPPrediction {
    symbol: String,
    prediction_interval: (f64, f64), // Lower and upper bounds
    point_prediction: f64,
    confidence_level: f64,
    temperature_scaling: f64,
    calibration_score: f64,
    prediction_horizon: u32, // minutes ahead
    timestamp: DateTime<Utc>,
}

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
struct PairIntelligence {
    symbol: String,
    market_data: MarketData,
    technical_indicators: TechnicalIndicators,
    whale_activity: WhaleActivity,
    ai_confidence: f64,
    profitability_score: f64,
    recommendation_score: f64,
    risk_level: String,
    last_updated: DateTime<Utc>,
    available_exchanges: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TechnicalIndicators {
    atr: f64,
    rsi: f64,
    volatility_score: f64,
    volume_profile: f64,
    fibonacci_level: Option<f64>,
    support_resistance: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WhaleActivity {
    large_orders_count: u32,
    whale_volume_ratio: f64,
    order_book_depth: f64,
    unusual_activity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CWTSAnalysis {
    pair_intelligence: PairIntelligence,
    ats_cp_predictions: Vec<ATSCPPrediction>,
    trading_signals: TradingSignals,
    risk_assessment: RiskAssessment,
    market_regime: MarketRegime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TradingSignals {
    signal: String, // "BUY", "SELL", "HOLD"
    strength: f64, // 0.0 to 1.0
    entry_price: Option<f64>,
    stop_loss: Option<f64>,
    take_profit: Option<f64>,
    position_size: f64, // Percentage of capital
    reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RiskAssessment {
    risk_score: f64, // 0.0 to 10.0
    max_drawdown_estimate: f64,
    value_at_risk: f64,
    expected_return: f64,
    sharpe_ratio: f64,
    volatility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketRegime {
    regime: String, // "TRENDING", "RANGING", "VOLATILE", "CALM"
    confidence: f64,
    regime_change_probability: f64,
    dominant_timeframe: String,
}

type AnalysisStore = Arc<DashMap<String, CWTSAnalysis>>;

#[derive(Clone)]
struct AppState {
    analysis_store: AnalysisStore,
    client: Client,
    data_server_url: String,
    intelligence_server_url: String,
}

#[derive(Deserialize)]
struct AnalysisQuery {
    symbol: String,
    timeframe: Option<String>,
    prediction_horizon: Option<u32>,
}

#[derive(Deserialize)]
struct PairListQuery {
    limit: Option<u32>,
    min_confidence: Option<f64>,
    sort_by: Option<String>,
}

async fn fetch_pair_intelligence(client: &Client, intelligence_url: &str, symbol: &str) -> Result<PairIntelligence, Box<dyn std::error::Error + Send + Sync>> {
    let url = format!("{}/api/pair-analysis?symbol={}", intelligence_url, symbol);
    
    let response = client.get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    
    if !response.status().is_success() {
        return Err(format!("Intelligence API error: {}", response.status()).into());
    }
    
    let intelligence: PairIntelligence = response.json().await?;
    Ok(intelligence)
}

async fn fetch_market_data(client: &Client, data_url: &str, symbol: &str) -> Result<Vec<MarketData>, Box<dyn std::error::Error + Send + Sync>> {
    let url = format!("{}/api/market-data?pair={}", data_url, symbol);
    
    let response = client.get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    
    if !response.status().is_success() {
        return Err(format!("Data API error: {}", response.status()).into());
    }
    
    let data: Vec<MarketData> = response.json().await?;
    Ok(data)
}

fn generate_ats_cp_predictions(intelligence: &PairIntelligence) -> Vec<ATSCPPrediction> {
    let base_price = intelligence.market_data.price;
    let volatility = intelligence.technical_indicators.volatility_score / 100.0;
    let confidence = intelligence.ai_confidence;
    
    // Generate predictions for different horizons (15min, 1h, 4h, 1d)
    let horizons = vec![15, 60, 240, 1440];
    
    horizons.into_iter().map(|horizon| {
        // Adaptive Temperature Scaling based on confidence
        let temperature_scaling = if confidence > 0.7 {
            0.8 // High confidence = lower temperature (tighter intervals)
        } else if confidence > 0.5 {
            1.0 // Medium confidence = standard temperature
        } else {
            1.2 // Low confidence = higher temperature (wider intervals)
        };
        
        // Time-adjusted volatility
        let time_factor = (horizon as f64 / 60.0).sqrt(); // Square root of time scaling
        let adjusted_volatility = volatility * time_factor * temperature_scaling;
        
        // Point prediction with drift based on technical indicators
        let drift = if intelligence.technical_indicators.rsi > 70.0 {
            -0.01 // Overbought, expect reversion
        } else if intelligence.technical_indicators.rsi < 30.0 {
            0.01 // Oversold, expect bounce
        } else {
            0.0 // Neutral
        };
        
        let point_prediction = base_price * (1.0 + drift);
        
        // Conformal prediction intervals (simplified)
        let z_score = 1.96; // 95% confidence
        let margin = base_price * adjusted_volatility * z_score;
        let lower_bound = point_prediction - margin;
        let upper_bound = point_prediction + margin;
        
        // Calibration score based on historical accuracy (simplified)
        let calibration_score = confidence * 0.95; // Assume some model uncertainty
        
        ATSCPPrediction {
            symbol: intelligence.symbol.clone(),
            prediction_interval: (lower_bound, upper_bound),
            point_prediction,
            confidence_level: 0.95,
            temperature_scaling,
            calibration_score,
            prediction_horizon: horizon,
            timestamp: Utc::now(),
        }
    }).collect()
}

fn generate_trading_signals(intelligence: &PairIntelligence, predictions: &[ATSCPPrediction]) -> TradingSignals {
    let current_price = intelligence.market_data.price;
    let rsi = intelligence.technical_indicators.rsi;
    let volatility = intelligence.technical_indicators.volatility_score;
    let whale_activity = intelligence.whale_activity.unusual_activity_score;
    let ai_confidence = intelligence.ai_confidence;
    
    // Get 1-hour prediction for signal generation
    let hour_prediction = predictions.iter()
        .find(|p| p.prediction_horizon == 60)
        .unwrap();
    
    let expected_return = (hour_prediction.point_prediction - current_price) / current_price;
    
    // Signal generation logic
    let (signal, strength, reason) = if expected_return > 0.02 && rsi < 70.0 && ai_confidence > 0.6 {
        ("BUY".to_string(), (ai_confidence * (expected_return * 10.0)).min(1.0), 
         format!("Strong upward prediction ({:.2}%) with good confidence", expected_return * 100.0))
    } else if expected_return < -0.02 && rsi > 30.0 && ai_confidence > 0.6 {
        ("SELL".to_string(), (ai_confidence * (expected_return.abs() * 10.0)).min(1.0),
         format!("Strong downward prediction ({:.2}%) with good confidence", expected_return * 100.0))
    } else {
        ("HOLD".to_string(), 0.1, "Insufficient confidence or expected return for action".to_string())
    };
    
    // Position sizing based on volatility and confidence
    let base_position = 0.1; // 10% base position
    let volatility_adjustment = (1.0 - volatility / 10.0).max(0.1);
    let confidence_adjustment = ai_confidence;
    let position_size = base_position * volatility_adjustment * confidence_adjustment;
    
    // Risk management levels
    let stop_loss = if signal == "BUY" {
        Some(current_price * 0.98) // 2% stop loss
    } else if signal == "SELL" {
        Some(current_price * 1.02) // 2% stop loss for short
    } else {
        None
    };
    
    let take_profit = if signal == "BUY" {
        Some(hour_prediction.point_prediction * 0.95) // Take 95% of predicted gain
    } else if signal == "SELL" {
        Some(hour_prediction.point_prediction * 1.05) // Take 95% of predicted decline
    } else {
        None
    };
    
    TradingSignals {
        signal,
        strength,
        entry_price: Some(current_price),
        stop_loss,
        take_profit,
        position_size,
        reason,
    }
}

fn assess_risk(intelligence: &PairIntelligence, predictions: &[ATSCPPrediction]) -> RiskAssessment {
    let volatility = intelligence.technical_indicators.volatility_score / 100.0;
    let whale_activity = intelligence.whale_activity.unusual_activity_score / 10.0;
    let ai_confidence = intelligence.ai_confidence;
    
    // Risk score (0-10, where 10 is highest risk)
    let base_risk = volatility * 5.0;
    let whale_risk = whale_activity * 2.0;
    let confidence_risk = (1.0 - ai_confidence) * 3.0;
    let risk_score = (base_risk + whale_risk + confidence_risk).min(10.0);
    
    // VaR estimation (simplified)
    let daily_prediction = predictions.iter()
        .find(|p| p.prediction_horizon == 1440)
        .unwrap();
    
    let expected_return = (daily_prediction.point_prediction - intelligence.market_data.price) / intelligence.market_data.price;
    let value_at_risk = volatility * 1.65; // 95% VaR
    
    // Sharpe ratio estimation
    let risk_free_rate = 0.05 / 365.0; // 5% annual risk-free rate
    let excess_return = expected_return - risk_free_rate;
    let sharpe_ratio = if volatility > 0.0 { excess_return / volatility } else { 0.0 };
    
    RiskAssessment {
        risk_score,
        max_drawdown_estimate: volatility * 2.5,
        value_at_risk,
        expected_return,
        sharpe_ratio,
        volatility,
    }
}

fn detect_market_regime(intelligence: &PairIntelligence) -> MarketRegime {
    let volatility = intelligence.technical_indicators.volatility_score;
    let change_24h = intelligence.market_data.change_24h.abs();
    let volume_profile = intelligence.technical_indicators.volume_profile;
    let whale_activity = intelligence.whale_activity.unusual_activity_score;
    
    let (regime, confidence) = if volatility > 7.0 && change_24h > 5.0 {
        ("VOLATILE".to_string(), 0.8)
    } else if change_24h > 3.0 && volume_profile > 7.0 {
        ("TRENDING".to_string(), 0.7)
    } else if volatility < 3.0 && change_24h < 2.0 {
        ("CALM".to_string(), 0.9)
    } else {
        ("RANGING".to_string(), 0.6)
    };
    
    let regime_change_probability = if whale_activity > 5.0 {
        0.3 // High whale activity suggests potential regime change
    } else {
        0.1
    };
    
    MarketRegime {
        regime,
        confidence,
        regime_change_probability,
        dominant_timeframe: "1H".to_string(),
    }
}

async fn analyze_pair(client: &Client, data_url: &str, intelligence_url: &str, symbol: &str) -> Result<CWTSAnalysis, Box<dyn std::error::Error + Send + Sync>> {
    // Fetch intelligence data
    let intelligence = fetch_pair_intelligence(client, intelligence_url, symbol).await?;
    
    // Generate ATS-CP predictions
    let predictions = generate_ats_cp_predictions(&intelligence);
    
    // Generate trading signals
    let trading_signals = generate_trading_signals(&intelligence, &predictions);
    
    // Assess risk
    let risk_assessment = assess_risk(&intelligence, &predictions);
    
    // Detect market regime
    let market_regime = detect_market_regime(&intelligence);
    
    Ok(CWTSAnalysis {
        pair_intelligence: intelligence,
        ats_cp_predictions: predictions,
        trading_signals,
        risk_assessment,
        market_regime,
    })
}

async fn get_pair_analysis(
    State(state): State<AppState>,
    Query(params): Query<AnalysisQuery>
) -> Result<Json<CWTSAnalysis>, StatusCode> {
    match analyze_pair(&state.client, &state.data_server_url, &state.intelligence_server_url, &params.symbol).await {
        Ok(analysis) => {
            // Cache the analysis
            state.analysis_store.insert(params.symbol.clone(), analysis.clone());
            Ok(Json(analysis))
        }
        Err(e) => {
            error!("Failed to analyze pair {}: {}", params.symbol, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_recommended_pairs(
    State(state): State<AppState>,
    Query(params): Query<PairListQuery>
) -> Result<Json<Vec<String>>, StatusCode> {
    // Fetch top recommendations from intelligence server
    let limit = params.limit.unwrap_or(10);
    let url = format!("{}/api/intelligence?limit={}&sort_by=recommendation_score", 
                     state.intelligence_server_url, limit);
    
    match state.client.get(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(intelligence_data) = response.json::<Vec<PairIntelligence>>().await {
                    let symbols: Vec<String> = intelligence_data.into_iter()
                        .filter(|p| p.ai_confidence >= params.min_confidence.unwrap_or(0.5))
                        .map(|p| p.symbol)
                        .collect();
                    return Ok(Json(symbols));
                }
            }
        }
        Err(e) => error!("Failed to fetch recommendations: {}", e),
    }
    
    Err(StatusCode::SERVICE_UNAVAILABLE)
}

async fn health_check() -> &'static str {
    "🎯 CWTS Core Server - Online"
}

async fn get_system_status(
    State(state): State<AppState>
) -> Json<HashMap<String, serde_json::Value>> {
    let cached_analyses = state.analysis_store.len();
    
    // Test connectivity to other services
    let data_server_healthy = state.client.get(&format!("{}/health", state.data_server_url))
        .send().await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    
    let intelligence_server_healthy = state.client.get(&format!("{}/health", state.intelligence_server_url))
        .send().await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    
    let mut status = HashMap::new();
    status.insert("cached_analyses".to_string(), serde_json::Value::Number(cached_analyses.into()));
    status.insert("data_server_connection".to_string(), serde_json::Value::Bool(data_server_healthy));
    status.insert("intelligence_server_connection".to_string(), serde_json::Value::Bool(intelligence_server_healthy));
    status.insert("last_updated".to_string(), serde_json::Value::String(Utc::now().to_rfc3339()));
    status.insert("status".to_string(), serde_json::Value::String("operational".to_string()));
    
    Json(status)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("cwts_core_server=info".parse()?))
        .init();
    
    info!("🎯 Starting CWTS Core Server...");
    
    // Initialize HTTP client
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent("CWTS-Core/0.1.0")
        .build()?;
    
    // Service URLs
    let data_server_url = "http://localhost:8010".to_string();
    let intelligence_server_url = "http://localhost:8021".to_string();
    
    // Initialize analysis cache
    let analysis_store = Arc::new(DashMap::new());
    
    let state = AppState {
        analysis_store,
        client,
        data_server_url,
        intelligence_server_url,
    };
    
    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/analyze", get(get_pair_analysis))
        .route("/api/recommendations", get(get_recommended_pairs))
        .route("/api/status", get(get_system_status))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(CorsLayer::permissive())
        )
        .with_state(state);
    
    // Start server
    let port = std::env::var("PORT").unwrap_or_else(|_| "8030".to_string());
    let addr = format!("0.0.0.0:{}", port);
    
    info!("🎯 CWTS Core Server starting on {}", addr);
    info!("📊 Endpoints:");
    info!("  • Health: http://{}/health", addr);
    info!("  • Analyze Pair: http://{}/api/analyze?symbol=BTCUSDT", addr);
    info!("  • Recommendations: http://{}/api/recommendations?limit=10", addr);
    info!("  • Status: http://{}/api/status", addr);
    info!("🔗 Integration:");
    info!("  • Data Server: http://localhost:8010");
    info!("  • Intelligence Server: http://localhost:8021");
    info!("🧠 Features: ATS-CP Predictions, Trading Signals, Risk Assessment, Market Regime Detection");
    
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}