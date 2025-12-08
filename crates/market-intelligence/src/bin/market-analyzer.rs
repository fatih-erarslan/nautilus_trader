//! Market Intelligence Analyzer Binary
//! 
//! Advanced market analysis using machine learning and sentiment analysis
//! for cryptocurrency market intelligence.

use std::env;
use std::path::Path;
use std::fs;
use serde_json;
use chrono::{DateTime, Utc};
use anyhow::{Result, anyhow};

#[derive(Debug)]
struct AnalyzerArgs {
    input_path: String,
    output_path: String,
    symbol: String,
    analysis_type: Vec<String>,
    lookback_days: i32,
}

impl AnalyzerArgs {
    fn parse() -> Result<Self> {
        let args: Vec<String> = env::args().collect();
        
        let mut input_path = String::new();
        let mut output_path = String::new();
        let mut symbol = "BTCUSDT".to_string();
        let mut analysis_type = vec!["trend".to_string(), "sentiment".to_string()];
        let mut lookback_days = 30;
        
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--input" => {
                    i += 1;
                    if i >= args.len() { return Err(anyhow!("Missing input path")); }
                    input_path = args[i].clone();
                },
                "--output" => {
                    i += 1;
                    if i >= args.len() { return Err(anyhow!("Missing output path")); }
                    output_path = args[i].clone();
                },
                "--symbol" => {
                    i += 1;
                    if i >= args.len() { return Err(anyhow!("Missing symbol")); }
                    symbol = args[i].clone();
                },
                "--analysis" => {
                    i += 1;
                    if i >= args.len() { return Err(anyhow!("Missing analysis type")); }
                    analysis_type = args[i].split(',').map(|s| s.to_string()).collect();
                },
                "--lookback" => {
                    i += 1;
                    if i >= args.len() { return Err(anyhow!("Missing lookback days")); }
                    lookback_days = args[i].parse().unwrap_or(30);
                },
                _ => {}
            }
            i += 1;
        }
        
        if input_path.is_empty() || output_path.is_empty() {
            return Err(anyhow!("Input and output paths are required"));
        }
        
        Ok(AnalyzerArgs {
            input_path,
            output_path,
            symbol,
            analysis_type,
            lookback_days,
        })
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct MarketData {
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    // Enhanced features from data pipeline
    sma_20: Option<f64>,
    sma_50: Option<f64>,
    rsi: Option<f64>,
    macd: Option<f64>,
    volatility: Option<f64>,
    atr: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
struct MarketIntelligence {
    timestamp: String,
    symbol: String,
    analysis_timestamp: String,
    
    // Trend Analysis
    trend_direction: String,
    trend_strength: f64,
    trend_confidence: f64,
    
    // Sentiment Analysis
    market_sentiment: String,
    sentiment_score: f64,
    fear_greed_index: f64,
    
    // Volume Analysis
    volume_profile: String,
    volume_strength: f64,
    unusual_volume: bool,
    
    // Pattern Recognition
    detected_patterns: Vec<String>,
    pattern_confidence: f64,
    
    // Risk Assessment
    volatility_regime: String,
    risk_score: f64,
    max_drawdown_risk: f64,
    
    // AI Predictions
    price_prediction_1h: f64,
    price_prediction_4h: f64,
    price_prediction_24h: f64,
    prediction_confidence: f64,
    
    // Market Regime
    market_regime: String,
    regime_stability: f64,
    
    // Action Recommendations
    recommended_action: String,
    confidence_level: f64,
    risk_reward_ratio: f64,
}

fn analyze_market_data(input_path: &str, output_path: &str, symbol: &str, analysis_types: &[String], lookback_days: i32) -> Result<()> {
    println!("🧠 Analyzing {} market intelligence from {}", symbol, input_path);
    
    // Read processed CSV file
    let mut reader = csv::Reader::from_path(input_path)?;
    let mut market_data: Vec<MarketData> = Vec::new();
    
    // Parse CSV rows
    for result in reader.deserialize() {
        match result {
            Ok(record) => {
                let data: MarketData = record;
                market_data.push(data);
            },
            Err(e) => {
                eprintln!("Warning: Failed to parse row: {}", e);
                continue;
            }
        }
    }
    
    println!("📊 Loaded {} records for intelligence analysis", market_data.len());
    
    if market_data.is_empty() {
        return Err(anyhow!("No valid data found in input file"));
    }
    
    // Perform market intelligence analysis
    let intelligence_data = perform_market_analysis(&market_data, symbol, analysis_types)?;
    
    println!("🎯 Generated {} intelligence insights", intelligence_data.len());
    
    // Write intelligence data to output JSON
    write_intelligence_json(&intelligence_data, output_path)?;
    
    println!("🧠 Market intelligence saved to {}", output_path);
    
    Ok(())
}

fn perform_market_analysis(data: &[MarketData], symbol: &str, analysis_types: &[String]) -> Result<Vec<MarketIntelligence>> {
    let mut intelligence = Vec::new();
    
    // Take the most recent data points for analysis
    let recent_data = &data[data.len().saturating_sub(100)..];
    let closes: Vec<f64> = recent_data.iter().map(|d| d.close).collect();
    let volumes: Vec<f64> = recent_data.iter().map(|d| d.volume).collect();
    
    // Generate intelligence for the latest data point
    if let Some(latest) = recent_data.last() {
        let current_price = latest.close;
        
        // Trend Analysis
        let (trend_direction, trend_strength) = analyze_trend(&closes);
        let trend_confidence = calculate_trend_confidence(&closes);
        
        // Sentiment Analysis (simplified)
        let (sentiment, sentiment_score, fear_greed) = analyze_market_sentiment(&closes, &volumes);
        
        // Volume Analysis
        let (volume_profile, volume_strength, unusual_volume) = analyze_volume(&volumes);
        
        // Pattern Recognition
        let (patterns, pattern_confidence) = detect_patterns(&closes);
        
        // Risk Assessment
        let (volatility_regime, risk_score, max_drawdown) = assess_risk(&closes);
        
        // AI Predictions (simplified ML-like predictions)
        let (pred_1h, pred_4h, pred_24h, pred_confidence) = predict_prices(current_price, &closes);
        
        // Market Regime Detection
        let (market_regime, regime_stability) = detect_market_regime(&closes, &volumes);
        
        // Action Recommendations
        let (action, confidence, risk_reward) = recommend_action(
            &trend_direction, trend_strength, sentiment_score, risk_score
        );
        
        let intelligence_item = MarketIntelligence {
            timestamp: latest.timestamp.clone(),
            symbol: symbol.to_string(),
            analysis_timestamp: Utc::now().to_rfc3339(),
            
            trend_direction,
            trend_strength,
            trend_confidence,
            
            market_sentiment: sentiment,
            sentiment_score,
            fear_greed_index: fear_greed,
            
            volume_profile,
            volume_strength,
            unusual_volume,
            
            detected_patterns: patterns,
            pattern_confidence,
            
            volatility_regime,
            risk_score,
            max_drawdown_risk: max_drawdown,
            
            price_prediction_1h: pred_1h,
            price_prediction_4h: pred_4h,
            price_prediction_24h: pred_24h,
            prediction_confidence: pred_confidence,
            
            market_regime,
            regime_stability,
            
            recommended_action: action,
            confidence_level: confidence,
            risk_reward_ratio: risk_reward,
        };
        
        intelligence.push(intelligence_item);
    }
    
    Ok(intelligence)
}

fn analyze_trend(closes: &[f64]) -> (String, f64) {
    if closes.len() < 20 {
        return ("UNKNOWN".to_string(), 0.0);
    }
    
    let recent_avg = closes[closes.len()-10..].iter().sum::<f64>() / 10.0;
    let older_avg = closes[closes.len()-20..closes.len()-10].iter().sum::<f64>() / 10.0;
    
    let trend_strength = ((recent_avg - older_avg) / older_avg).abs();
    
    let direction = if recent_avg > older_avg * 1.02 {
        "BULLISH"
    } else if recent_avg < older_avg * 0.98 {
        "BEARISH"
    } else {
        "SIDEWAYS"
    };
    
    (direction.to_string(), trend_strength.min(1.0))
}

fn calculate_trend_confidence(closes: &[f64]) -> f64 {
    if closes.len() < 10 {
        return 0.5;
    }
    
    // Calculate trend consistency
    let mut consistent_moves = 0;
    for window in closes.windows(2) {
        if window[1] > window[0] {
            consistent_moves += 1;
        }
    }
    
    consistent_moves as f64 / (closes.len() - 1) as f64
}

fn analyze_market_sentiment(closes: &[f64], volumes: &[f64]) -> (String, f64, f64) {
    let price_momentum = if closes.len() >= 5 {
        (closes[closes.len()-1] - closes[closes.len()-5]) / closes[closes.len()-5]
    } else {
        0.0
    };
    
    let volume_momentum = if volumes.len() >= 5 {
        let recent_vol = volumes[volumes.len()-5..].iter().sum::<f64>() / 5.0;
        let older_vol = volumes[volumes.len().saturating_sub(10)..volumes.len()-5].iter().sum::<f64>() / 5.0;
        if older_vol > 0.0 { (recent_vol - older_vol) / older_vol } else { 0.0 }
    } else {
        0.0
    };
    
    let sentiment_score = (price_momentum + volume_momentum * 0.3).tanh(); // Normalize to [-1, 1]
    
    let sentiment = if sentiment_score > 0.3 {
        "BULLISH"
    } else if sentiment_score < -0.3 {
        "BEARISH"
    } else {
        "NEUTRAL"
    };
    
    let fear_greed = ((sentiment_score + 1.0) / 2.0) * 100.0; // Convert to 0-100 scale
    
    (sentiment.to_string(), sentiment_score, fear_greed)
}

fn analyze_volume(volumes: &[f64]) -> (String, f64, bool) {
    if volumes.len() < 10 {
        return ("UNKNOWN".to_string(), 0.5, false);
    }
    
    let recent_avg = volumes[volumes.len()-5..].iter().sum::<f64>() / 5.0;
    let historical_avg = volumes.iter().sum::<f64>() / volumes.len() as f64;
    
    let volume_strength = (recent_avg / historical_avg).min(3.0) / 3.0; // Normalize
    let unusual_volume = recent_avg > historical_avg * 2.0;
    
    let profile = if volume_strength > 0.8 {
        "HIGH"
    } else if volume_strength > 0.4 {
        "MODERATE"
    } else {
        "LOW"
    };
    
    (profile.to_string(), volume_strength, unusual_volume)
}

fn detect_patterns(closes: &[f64]) -> (Vec<String>, f64) {
    let mut patterns = Vec::new();
    let mut total_confidence = 0.0;
    
    if closes.len() >= 20 {
        // Simple pattern detection
        let recent = &closes[closes.len()-10..];
        
        // Check for breakout pattern
        if recent.iter().all(|&p| p > closes[closes.len()-20]) {
            patterns.push("BREAKOUT_BULLISH".to_string());
            total_confidence += 0.7;
        }
        
        // Check for support/resistance
        let max_price = recent.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_price = recent.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if (max_price - min_price) / min_price < 0.02 { // Less than 2% range
            patterns.push("CONSOLIDATION".to_string());
            total_confidence += 0.6;
        }
    }
    
    if patterns.is_empty() {
        patterns.push("NO_CLEAR_PATTERN".to_string());
        total_confidence = 0.3;
    }
    
    let avg_confidence = total_confidence / patterns.len() as f64;
    (patterns, avg_confidence.min(1.0))
}

fn assess_risk(closes: &[f64]) -> (String, f64, f64) {
    if closes.len() < 20 {
        return ("UNKNOWN".to_string(), 0.5, 0.1);
    }
    
    // Calculate volatility
    let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0] - 1.0).powi(2)).collect();
    let volatility = (returns.iter().sum::<f64>() / returns.len() as f64).sqrt();
    
    let volatility_regime = if volatility > 0.05 {
        "HIGH_VOLATILITY"
    } else if volatility > 0.02 {
        "MODERATE_VOLATILITY"
    } else {
        "LOW_VOLATILITY"
    };
    
    let risk_score = (volatility * 10.0).min(1.0); // Normalize to 0-1
    
    // Calculate max drawdown potential
    let max_price = closes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let current_price = closes[closes.len()-1];
    let max_drawdown = ((max_price - current_price) / max_price).max(0.0);
    
    (volatility_regime.to_string(), risk_score, max_drawdown)
}

fn predict_prices(current_price: f64, closes: &[f64]) -> (f64, f64, f64, f64) {
    if closes.len() < 10 {
        return (current_price, current_price, current_price, 0.3);
    }
    
    // Simple trend-based predictions
    let short_trend = (closes[closes.len()-1] - closes[closes.len()-3]) / closes[closes.len()-3];
    let medium_trend = (closes[closes.len()-1] - closes[closes.len()-7]) / closes[closes.len()-7];
    let long_trend = (closes[closes.len()-1] - closes[closes.len()-10]) / closes[closes.len()-10];
    
    let pred_1h = current_price * (1.0 + short_trend * 0.5);
    let pred_4h = current_price * (1.0 + medium_trend * 0.8);
    let pred_24h = current_price * (1.0 + long_trend * 1.2);
    
    let confidence = 0.6; // Moderate confidence in simple predictions
    
    (pred_1h, pred_4h, pred_24h, confidence)
}

fn detect_market_regime(closes: &[f64], volumes: &[f64]) -> (String, f64) {
    if closes.len() < 20 {
        return ("UNKNOWN".to_string(), 0.5);
    }
    
    let volatility = calculate_volatility(closes);
    let trend_strength = calculate_trend_strength(closes);
    let volume_consistency = calculate_volume_consistency(volumes);
    
    let regime = if volatility > 0.04 && trend_strength > 0.3 {
        "TRENDING_VOLATILE"
    } else if volatility < 0.02 && trend_strength < 0.1 {
        "RANGING_STABLE"
    } else if trend_strength > 0.2 {
        "TRENDING_STABLE"
    } else {
        "RANGING_VOLATILE"
    };
    
    let stability = (volume_consistency + (1.0 - volatility) * 0.5) / 1.5;
    
    (regime.to_string(), stability.min(1.0))
}

fn calculate_volatility(closes: &[f64]) -> f64 {
    let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0] - 1.0).powi(2)).collect();
    (returns.iter().sum::<f64>() / returns.len() as f64).sqrt()
}

fn calculate_trend_strength(closes: &[f64]) -> f64 {
    if closes.len() < 10 {
        return 0.0;
    }
    
    let start = closes[0];
    let end = closes[closes.len()-1];
    ((end - start) / start).abs()
}

fn calculate_volume_consistency(volumes: &[f64]) -> f64 {
    if volumes.len() < 10 {
        return 0.5;
    }
    
    let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
    let variance = volumes.iter().map(|v| (v - avg_volume).powi(2)).sum::<f64>() / volumes.len() as f64;
    let std_dev = variance.sqrt();
    
    1.0 - (std_dev / avg_volume).min(1.0)
}

fn recommend_action(trend: &str, trend_strength: f64, sentiment_score: f64, risk_score: f64) -> (String, f64, f64) {
    let bullish_signal = trend == "BULLISH" && sentiment_score > 0.2;
    let bearish_signal = trend == "BEARISH" && sentiment_score < -0.2;
    
    let action = if bullish_signal && risk_score < 0.7 {
        "BUY"
    } else if bearish_signal && risk_score < 0.7 {
        "SELL"
    } else if risk_score > 0.8 {
        "WAIT"
    } else {
        "HOLD"
    };
    
    let confidence = (trend_strength + sentiment_score.abs()) / 2.0 * (1.0 - risk_score * 0.5);
    let risk_reward = if risk_score > 0.5 { 1.0 / risk_score } else { 2.0 };
    
    (action.to_string(), confidence.min(1.0), risk_reward.min(5.0))
}

fn write_intelligence_json(data: &[MarketIntelligence], output_path: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(data)?;
    fs::write(output_path, json)?;
    Ok(())
}

fn main() -> Result<()> {
    println!("🧠 SUPERMAN MARKET INTELLIGENCE ANALYZER");
    println!("=========================================");
    
    let args = AnalyzerArgs::parse()?;
    
    println!("📝 Configuration:");
    println!("   Input:        {}", args.input_path);
    println!("   Output:       {}", args.output_path);
    println!("   Symbol:       {}", args.symbol);
    println!("   Analysis:     {:?}", args.analysis_type);
    println!("   Lookback:     {} days", args.lookback_days);
    
    // Verify input file exists
    if !Path::new(&args.input_path).exists() {
        return Err(anyhow!("Input file does not exist: {}", args.input_path));
    }
    
    // Perform market intelligence analysis
    analyze_market_data(&args.input_path, &args.output_path, &args.symbol, &args.analysis_type, args.lookback_days)?;
    
    println!("🎯 Market intelligence analysis completed successfully!");
    
    Ok(())
}